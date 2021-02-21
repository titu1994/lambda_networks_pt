import torch
import pytest

from lambda_networks import lambda_resnets


DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_18(device):
    input_dim = 224
    model = lambda_resnets.resnet18(lambda_m=True, input_size=input_dim)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 8.8e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_18_local_context(device):
    model = lambda_resnets.resnet18(lambda_r=7)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 8.8e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_34(device):
    input_dim = 224
    model = lambda_resnets.resnet34(lambda_m=True, input_size=input_dim)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 16.5e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_34_local_context(device):
    model = lambda_resnets.resnet34(lambda_r=7)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 16.5e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_50(device):
    input_dim = 224
    model = lambda_resnets.resnet50(lambda_m=True, input_size=input_dim)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 20.5e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_50_local_context(device):
    model = lambda_resnets.resnet50(lambda_r=7)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 20.5e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_101(device):
    input_dim = 224
    model = lambda_resnets.resnet101(lambda_m=True, input_size=input_dim)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 38.4e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_101_local_context(device):
    model = lambda_resnets.resnet101(lambda_r=7)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 38.3e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_152(device):
    input_dim = 224
    model = lambda_resnets.resnet152(lambda_m=True, input_size=input_dim)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 52.4e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


@pytest.mark.parametrize("device", DEVICES)
def test_resnet_152_local_context(device):
    model = lambda_resnets.resnet152(lambda_r=7)
    model = model.to(device)

    num_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert num_parameters > 52.3e6

    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])


if __name__ == '__main__':
    pytest.main([__file__])
