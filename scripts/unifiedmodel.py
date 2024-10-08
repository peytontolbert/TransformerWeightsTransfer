import torch.nn as nn
from models.transformer import SimpleTransformer
from models.mamba import MambaModel
from models.lstm import SimpleLSTM
from models.s4model import LiquidS4

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Linear(input_dim, projection_dim)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])  # Flatten for projection
        return self.projection(x)

class UnifiedModel(nn.Module):
    def __init__(self, input_dim, shared_embedding_dim, projection_dim, model_params):
        super(UnifiedModel, self).__init__()
        # Shared Embedding Layer
        self.shared_embedding = nn.Linear(input_dim, shared_embedding_dim)

        # Individual Models
        self.transformer = SimpleTransformer(
            input_dim=shared_embedding_dim,
            model_dim=model_params['transformer']['model_dim'],
            output_dim=model_params['transformer']['output_dim'],
            num_heads=model_params['transformer']['num_heads'],
            num_layers=model_params['transformer']['num_layers']
        )
        self.mamba = MambaModel(
            input_dim=shared_embedding_dim,
            hidden_dim=model_params['mamba']['hidden_dim'],  # Ensure this is set to 1
            output_dim=model_params['mamba']['output_dim'],
            num_layers=model_params['mamba']['num_layers']
        )
        self.lstm = SimpleLSTM(
            input_dim=shared_embedding_dim,
            hidden_dim=model_params['lstm']['hidden_dim'],
            output_dim=model_params['lstm']['output_dim'],
            num_layers=model_params['lstm']['num_layers']
        )
        self.liquid_s4 = LiquidS4(
            state_dim=model_params['liquid_s4']['state_dim'],
            input_dim=shared_embedding_dim,
            output_dim=model_params['liquid_s4']['output_dim'],
            liquid_order=model_params['liquid_s4']['liquid_order']
        )

        # Feature Projections for Intermediate Features
        # Ensure projection_dim matches the feature output dimension
        self.feature_projections = nn.ModuleList([
            FeatureProjection(model_params['transformer']['model_dim'], projection_dim),
            FeatureProjection(model_params['mamba']['hidden_dim'], projection_dim),  # Should be 1
            FeatureProjection(model_params['lstm']['hidden_dim'], projection_dim),
            FeatureProjection(model_params['liquid_s4']['state_dim'], projection_dim)
        ])

        # Initialize all sub-models
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize shared embedding
        nn.init.xavier_uniform_(self.shared_embedding.weight)
        if self.shared_embedding.bias is not None:
            nn.init.zeros_(self.shared_embedding.bias)
        
        # Initialize individual models
        for name, module in self.named_children():
            if isinstance(module, (SimpleTransformer, MambaModel, SimpleLSTM, LiquidS4)):
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)

    def forward(self, x):
        # print("Input shape:", x.shape)  # Debug: Input shape

        # Shared Embedding
        embedded = self.shared_embedding(x)
        # print("Embedded shape:", embedded.shape)  # Debug: Embedded shape

        # Individual Model Outputs and Features
        transformer_output, transformer_features = self.transformer(embedded)
        # print("Transformer output shape:", transformer_output.shape)  # Debug: Transformer output shape
        # print("Transformer features shape:", transformer_features.shape)  # Debug: Transformer features shape

        mamba_output, mamba_features = self.mamba(embedded)
        # print("Mamba output shape:", mamba_output.shape)  # Debug: Mamba output shape
        # print("Mamba features shape:", mamba_features.shape)  # Debug: Mamba features shape

        lstm_output, lstm_features = self.lstm(embedded)
        # print("LSTM output shape:", lstm_output.shape)  # Debug: LSTM output shape
        # print("LSTM features shape:", lstm_features.shape)  # Debug: LSTM features shape

        liquid_output, liquid_features = self.liquid_s4(embedded)
        # print("Liquid S4 output shape:", liquid_output.shape)  # Debug: Liquid S4 output shape
        # print("Liquid S4 features shape:", liquid_features.shape)  # Debug: Liquid S4 features shape

        outputs = [transformer_output, mamba_output, lstm_output, liquid_output]
        features = [transformer_features, mamba_features, lstm_features, liquid_features]
        #for i, feat in enumerate(features):
            # print(f"Feature {i} shape: {feat.shape}")
        #for i, output in enumerate(outputs):
            # print(f"Output {i} shape: {output.shape}")

        # Add assertion to ensure projection layers match feature dimensions
        for i, (proj, feat) in enumerate(zip(self.feature_projections, features)):
            assert proj.projection.in_features == feat.shape[-1], \
                f"Projection layer {i} expects input dim {proj.projection.in_features}, but got {feat.shape[-1]}"

        # Project Features to Common Latent Space
        projected_features = [proj(feat) for proj, feat in zip(self.feature_projections, features)]
        #for i, proj_feat in enumerate(projected_features):
            # print(f"Projected features {i} shape:", proj_feat.shape)  # Debug: Projected features shape

        return outputs, projected_features