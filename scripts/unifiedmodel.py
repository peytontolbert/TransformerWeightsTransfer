import torch
import torch.nn as nn
from models.transformer import SimpleTransformer
from models.mamba import MambaModel
from models.lstm import SimpleLSTM
from models.s4model import LiquidS4

class FeatureProjection(nn.Module):
    def __init__(self, in_features, projection_dim):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Linear(in_features, projection_dim)

    def forward(self, x):
        return self.projection(x)

class UnifiedModel(nn.Module):
    def __init__(self, input_dim, shared_embedding_dim, projection_dim, model_params, dropout_prob=0.01):
        super(UnifiedModel, self).__init__()
        # Shared Embedding Layer
        self.shared_embedding = nn.Linear(input_dim, shared_embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)  # Apply dropout after shared embedding

        # Individual Models
        self.transformer = SimpleTransformer(
            input_dim=shared_embedding_dim,
            model_dim=model_params['transformer']['model_dim'],
            output_dim=model_params['transformer']['output_dim'],
            num_heads=model_params['transformer']['num_heads'],
            num_layers=model_params['transformer']['num_layers'],
            dropout_prob=dropout_prob
        )
        self.mamba = MambaModel(
            input_dim=shared_embedding_dim,
            hidden_dim=model_params['mamba']['hidden_dim'],
            output_dim=model_params['mamba']['output_dim'],
            num_layers=model_params['mamba']['num_layers'],
            dropout_prob=dropout_prob
        )
        self.lstm = SimpleLSTM(
            input_dim=shared_embedding_dim,
            hidden_dim=model_params['lstm']['hidden_dim'],
            output_dim=model_params['lstm']['output_dim'],
            num_layers=model_params['lstm']['num_layers'],
            dropout_prob=dropout_prob
        )
        self.liquid_s4 = LiquidS4(
            state_dim=model_params['liquid_s4']['state_dim'],
            input_dim=shared_embedding_dim,
            output_dim=model_params['liquid_s4']['output_dim'],
            liquid_order=model_params['liquid_s4']['liquid_order'],
            dropout_prob=dropout_prob
        )

        # Initialize Feature Projections dynamically based on sub-models' output dimensions
        self.projections = nn.ModuleList([
            FeatureProjection(model_params['transformer']['output_dim'], projection_dim),
            FeatureProjection(model_params['mamba']['output_dim'], projection_dim),
            FeatureProjection(model_params['lstm']['output_dim'], projection_dim),
            FeatureProjection(model_params['liquid_s4']['output_dim'], projection_dim)
        ])

        # Initialize all sub-models
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    """
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
    """
    def forward(self, x):
        # print("Input shape:", x.shape)  # Debug: Input shape

        # Shared Embedding
        embedded = self.shared_embedding(x)
        # print("Embedded shape:", embedded.shape)  # Debug: Embedded shape
        embedded = self.dropout(embedded)  # Apply dropout to the shared embedding

        # Individual Model Outputs and Features
        transformer_output, transformer_features = self.transformer(embedded)
        #print("Transformer output shape:", transformer_output.shape)  # Debug: Transformer output shape
        #print("Transformer features shape:", transformer_features.shape)  # Debug: Transformer features shape

        mamba_output, mamba_features = self.mamba(embedded)
        #print("Mamba output shape:", mamba_output.shape)  # Debug: Mamba output shape
        #print("Mamba features shape:", mamba_features.shape)  # Debug: Mamba features shape

        lstm_output, lstm_features = self.lstm(embedded)
        #print("LSTM output shape:", lstm_output.shape)  # Debug: LSTM output shape
        #print("LSTM features shape:", lstm_features.shape)  # Debug: LSTM features shape

        liquid_output, liquid_features = self.liquid_s4(embedded)
        #print("Liquid S4 output shape:", liquid_output.shape)  # Debug: Liquid S4 output shape
        #print("Liquid S4 features shape:", liquid_features.shape)  # Debug: Liquid S4 features shape

        outputs = [transformer_output, mamba_output, lstm_output, liquid_output]
        features = [transformer_features, mamba_features, lstm_features, liquid_features]
        #for i, feat in enumerate(features):
            # print(f"Feature {i} shape: {feat.shape}")
        #for i, output in enumerate(outputs):
            # print(f"Output {i} shape: {output.shape}")

        # Apply projection layers
        projected_features = []
        for i, proj in enumerate(self.projections):
            feat = outputs[i]
            #print(f"Projection layer {i}: expected in_features={proj.projection.in_features}, got feat.shape={feat.shape}")
            assert proj.projection.in_features == feat.shape[-1], \
                f"Projection layer {i} expects input dim {proj.projection.in_features}, but got {feat.shape[-1]}"
            projected_feat = proj(feat)
            projected_features.append(projected_feat)

        # Combine projected features as needed (e.g., concatenation, averaging)
        # Example: Concatenation
        combined_features = torch.cat(projected_features, dim=-1)

        return outputs, projected_features