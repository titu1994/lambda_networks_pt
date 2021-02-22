import torch
import pytest

from lambda_networks import lambda_module_3d


DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra", [1, 4])
def test_construction(device, dim_intra):
    layer = lambda_module_3d.LambdaLayer3D(64, 64, dim_k=16, m=8, heads=4, dim_intra=dim_intra)
    layer = layer.to(device)

    x = torch.zeros(5, 64, 8, 8, 8, device=device)
    out = layer(x)

    assert out.shape == (5, 64, 8, 8, 8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra", [1, 4])
def test_wrong_m(device, dim_intra):
    layer = lambda_module_3d.LambdaLayer3D(12, 12, dim_k=16, m=4, heads=4, dim_intra=dim_intra)
    layer = layer.to(device)

    # m in layer = 4, m in input = 8
    x = torch.zeros(5, 12, 8, 8, 8, device=device)

    with pytest.raises(RuntimeError):
        _ = layer(x)


@pytest.mark.parametrize("dim_intra", [1, 4])
def test_construction_local_context_impl_0(dim_intra):
    with pytest.raises(AssertionError):
        _ = lambda_module_3d.LambdaLayer3D(
            64, 64, dim_k=16, r=7, heads=4, dim_intra=dim_intra, implementation=0
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_intra", [1, 4])
def test_construction_local_context_impl_1(device, dim_intra):
    layer = lambda_module_3d.LambdaLayer3D(
        64, 64, dim_k=16, r=7, heads=4, dim_intra=dim_intra, implementation=1
    )
    layer = layer.to(device)

    x = torch.zeros(5, 64, 8, 8, 8, device=device)
    out = layer(x)

    assert out.shape == (5, 64, 8, 8, 8)


if __name__ == '__main__':
    pytest.main([__file__])
