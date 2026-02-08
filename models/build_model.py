from models.models import GATTransformerNodeOnly, DeepLeapTransModel, EHEvolverSandwichModel
from models.gnn_MetaLayers import PhysicsAugEGATModel
from models.gnn_models import GATConvNetworkNodeOnly, DeepLeapfrogModel
from models.transformer import TransformerNodeRegressor

def build_model(model_name: str, in_dim: int = 9, edge_dim: int = 4, transformer_mode: str = "mask"):
    name = (model_name or '').lower()
    if name == 'gat_transformer':
        return GATTransformerNodeOnly(
            out_vertex_dim=3,
            in_vertex_dim=in_dim,
            edge_dim=edge_dim,
            transformer_mode=transformer_mode,
        )
    if name == 'gat_conv':
        return GATConvNetworkNodeOnly(out_vertex_dim=3, in_vertex_dim=in_dim, edge_dim=edge_dim)
    if name == 'transformer':
        return TransformerNodeRegressor(in_dim=in_dim, out_dim=3, mode=transformer_mode)
    if name == 'deepleapfrog':
        return DeepLeapfrogModel()
    if name == 'deepleaptrans':
        return DeepLeapTransModel()
    if name == 'ehevolver':
        return EHEvolverSandwichModel()
    if name == 'phys_egat':
        return PhysicsAugEGATModel()
 
    


def build_model_from_loader(args, loader):
    """Infer feature dimensions from the loader's first batch and build the model."""
    model_name = args.get('model', 'gat_conv')
    name_lower = (model_name or '').lower()
    if name_lower in {'deepleapfrog', 'deepleaptrans'}:
        return build_model(model_name)

    sample_batch = next(iter(loader), None)
    if sample_batch is None:
        raise RuntimeError("Empty training data; cannot infer input dimensions")

    sample = sample_batch
    while isinstance(sample, (list, tuple)) and len(sample) > 0:
        sample = sample[0]

    data0 = sample
    if not hasattr(data0, 'x'):
        raise RuntimeError(f"Cannot infer input dims from batch element of type {type(data0)}; expected PyG Data/Batch.")

    in_dim = data0.x.shape[1]
    edge_dim = data0.edge_attr.shape[1] if hasattr(data0, 'edge_attr') and data0.edge_attr is not None else 4
    transformer_mode = args.get('transformer_mode', args.get('transformer-mode', 'mask')) if name_lower in {'transformer', 'gat_transformer'} else 'mask'
    return build_model(model_name, in_dim=in_dim, edge_dim=edge_dim, transformer_mode=transformer_mode)