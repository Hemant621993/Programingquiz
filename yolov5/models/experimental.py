def attempt_load(weights, device):
    """Mock function for loading YOLOv5 model"""
    class MockModel:
        def __init__(self):
            pass
            
        def __call__(self, x):
            # Return a mock prediction tensor
            import torch
            return [torch.zeros((1, 10, 6), device=device)]  # shape: [batch_size, num_detections, 6]
            
        def to(self, device):
            return self
            
    return MockModel()