import torch
import pytest

from lambda_networks import lambda_module_2d


DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra_dimension", [1, 4])
def test_construction(device, dim_intra_dimension):
    layer = lambda_module_2d.LambdaLayer2D(64, 64, dim_k=16, m=32, heads=4, dim_intra_dimension=dim_intra_dimension)
    layer = layer.to(device)

    x = torch.zeros(5, 64, 32, 32, device=device)
    out = layer(x)

    assert out.shape == (5, 64, 32, 32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra_dimension", [1, 4])
def test_wrong_m(device, dim_intra_dimension):
    layer = lambda_module_2d.LambdaLayer2D(64, 64, dim_k=16, m=64, heads=4, dim_intra_dimension=dim_intra_dimension)
    layer = layer.to(device)

    # m in layer = 64, m in input = 32
    x = torch.zeros(5, 64, 32, 32, device=device)

    with pytest.raises(RuntimeError):
        _ = layer(x)

    layer = lambda_module_2d.LambdaLayer2D(64, 64, dim_k=16, m=16, heads=4, dim_intra_dimension=dim_intra_dimension)
    layer = layer.to(device)

    # m in layer = 64, m in input = 16
    x = torch.zeros(5, 64, 32, 32, device=device)

    with pytest.raises(RuntimeError):
        _ = layer(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra_dimension", [1, 4])
def test_construction_local_context_impl_0(device, dim_intra_dimension):
    layer = lambda_module_2d.LambdaLayer2D(
        64, 64, dim_k=16, r=7, heads=4, dim_intra_dimension=dim_intra_dimension, implementation=0
    )
    layer = layer.to(device)

    x = torch.zeros(5, 64, 32, 32, device=device)
    out = layer(x)

    assert out.shape == (5, 64, 32, 32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra_dimension", [1, 4])
def test_construction_local_context_impl_1(device, dim_intra_dimension):
    layer = lambda_module_2d.LambdaLayer2D(
        64, 64, dim_k=16, r=7, heads=4, dim_intra_dimension=dim_intra_dimension, implementation=1
    )
    layer = layer.to(device)

    x = torch.zeros(5, 64, 32, 32, device=device)
    out = layer(x)

    assert out.shape == (5, 64, 32, 32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra_dimension", [1, 4])
def test_construction_local_context_equal_impl(device, dim_intra_dimension):
    layer0 = lambda_module_2d.LambdaLayer2D(
        16, 32, dim_k=16, r=5, heads=4, dim_intra_dimension=dim_intra_dimension, implementation=0
    )
    # Set weights explicitly
    layer0.to_k.weight.data = torch.ones_like(layer0.to_k.weight)
    layer0.to_q.weight.data = torch.ones_like(layer0.to_q.weight)
    layer0.to_v.weight.data = torch.ones_like(layer0.to_v.weight)
    layer0.pos_conv.weight.data = torch.ones_like(layer0.pos_conv.weight)
    layer0.pos_conv.bias.data = torch.ones_like(layer0.pos_conv.bias)

    layer1 = lambda_module_2d.LambdaLayer2D(
        16, 32, dim_k=16, r=5, heads=4, dim_intra_dimension=dim_intra_dimension, implementation=1
    )
    layer1.to_k.weight.data = torch.ones_like(layer1.to_k.weight)
    layer1.to_q.weight.data = torch.ones_like(layer1.to_q.weight)
    layer1.to_v.weight.data = torch.ones_like(layer1.to_v.weight)
    layer1.pos_conv.weight.data = torch.ones_like(layer1.pos_conv.weight)
    layer1.pos_conv.bias.data = torch.ones_like(layer1.pos_conv.bias)

    layer0 = layer0.to(device)
    layer1 = layer1.to(device)

    x = torch.randn(5, 16, 64, 64, device=device)
    out0 = layer0(x.clone())
    out1 = layer1(x.clone())

    diff = out0 - out1

    mse_tol = 0.0 if device == 'cpu' else 1e-8
    mae_tol = 0.0 if device == 'cpu' else 1e-4
    assert diff.square().mean().item() <= mse_tol
    assert diff.abs().mean().item() <= mae_tol


if __name__ == '__main__':
    pytest.main([__file__])
