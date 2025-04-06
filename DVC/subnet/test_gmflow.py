import torch
import unittest
from gmflow.gmflowclass.GMFL  import GMFlowEstimator  # Absolute import
from PIL import Image
import numpy as np
import os 
from gmflow.utils.flow_viz import save_vis_flow_tofile
import torch
import unittest

from PIL import Image
import numpy as np
import os
import time
import torch.cuda as cuda

class TestGMFlowEstimator(unittest.TestCase):
    def setUp(self):
        # Set up GMFlowEstimator instance with default parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume_path = './gmflow/pretrained/gmflow_things-e9887eda.pth'
        if not os.path.exists(self.resume_path):
            self.skipTest(f"Pretrained checkpoint not found at {self.resume_path}")

        self.model = GMFlowEstimator(
            device=self.device,
            resume=self.resume_path,
            pred_bidir_flow=False,  # Unidirectional flow
            fwd_bwd_consistency_check=False,
            padding_factor=16,
            inference_size=None
        )
        self.model.eval()  # Frozen mode
        for param in self.model.parameters():
            param.requires_grad = False  # Explicitly freeze

        # Image paths for testing (use your own if available)
        self.img1_path = './gmflow/demo/sintel_market_1/img1_batch.png'
        self.img2_path = './gmflow/demo/sintel_market_1/img2_batch.png'
        if not os.path.exists(self.img1_path) or not os.path.exists(self.img2_path):
            self.skipTest(f"Image files not found: {self.img1_path}, {self.img2_path}")

    def load_image(self, img_path):
        """Helper to load and preprocess an image."""
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        return img_tensor.to(self.device)

    def test_single_pair_speed(self):
        """Test forward pass speed for a single image pair."""
        img1 = self.load_image(self.img1_path)
        img2 = self.load_image(self.img2_path)

        # Warm-up run (to avoid initialization overhead)
        with torch.no_grad():
            _ = self.model.forward(img1, img2)

        # Measure time
        num_runs = 10
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):# cool
                flow = self.model.forward(img1, img2)
        avg_time = (time.time() - start_time) / num_runs
        print(f"Average forward pass time (single pair): {avg_time:.4f} seconds")
        self.assertTrue(avg_time < 5.0, f"Single pair forward pass too slow: {avg_time:.4f} seconds")

    def test_batch_speed(self):
        """Test forward pass speed for a batch of image pairs."""
        batch_size = 2 # Simulate a small batch; adjust based on your needs
        img1_batch = torch.cat([self.load_image(self.img1_path) for _ in range(batch_size)], dim=0)
        img2_batch = torch.cat([self.load_image(self.img2_path) for _ in range(batch_size)], dim=0)

        # Warm-up run
        with torch.no_grad():
            _ = self.model.forward(img1_batch, img2_batch)

        # Measure time
        num_runs = 5
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                flow = self.model.forward(img1_batch, img2_batch)
        avg_time = (time.time() - start_time) / num_runs
        avg_time_per_pair = avg_time / batch_size
        print(f"Average forward pass time (batch size {batch_size}): {avg_time:.4f} seconds")
        print(f"Average time per pair: {avg_time_per_pair:.4f} seconds")
        self.assertTrue(avg_time_per_pair < 5.0, f"Batch forward pass per pair too slow: {avg_time_per_pair:.4f} seconds")

    def test_memory_usage(self):
        """Test GPU memory usage for a single pair."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory test")

        img1 = self.load_image(self.img1_path)
        img2 = self.load_image(self.img2_path)

        # Reset peak memory stats and synchronize
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize()

        with torch.no_grad():
            flow = self.model.forward(img1, img2)

        # Measure peak memory after forward pass
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # Convert to GB
        print(f"Peak GPU memory usage (single pair): {peak_memory:.2f} GB")
        self.assertTrue(peak_memory < 8.0, f"Memory usage too high: {peak_memory:.2f} GB")

    def test_output_and_save(self):
        """Test output correctness and save flow visualization (same as original)."""
        img1 = self.load_image(self.img1_path)
        img2 = self.load_image(self.img2_path)
        output_flow_path = './gmflow/flow_output_test.png'

        with torch.no_grad():
            flow = self.model.forward(img1, img2)

        # Check output shape (assuming flow is a tensor; adjust if it's NumPy)
        self.assertEqual(flow.shape[0], 1, "Batch dimension should be 1")
        self.assertEqual(flow.shape[1], 2, "Flow should have 2 channels (x, y)")
        self.assertEqual(flow.shape[2], img1.shape[2], "Flow height should match input height")
        self.assertEqual(flow.shape[3], img1.shape[3], "Flow width should match input width")

        # Convert flow to NumPy for visualization
        if isinstance(flow, torch.Tensor):
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:  # If flow is already NumPy (as in your case)
            flow_np = flow[0].transpose(1, 2, 0)

        save_vis_flow_tofile(flow_np, output_flow_path)
        self.assertTrue(os.path.exists(output_flow_path), f"Flow visualization file {output_flow_path} not created")
        
  
    def tearDown(self):
        # Clean up VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()





# class TestGMFlowEstimator(unittest.TestCase):
#     def setUp(self):
#         # Set up a GMFlowEstimator instance with default parameters
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.resume_path = './gmflow/pretrained/gmflow_things-e9887eda.pth'  # Adjust path as needed
#         if not os.path.exists(self.resume_path):
#             self.skipTest(f"Pretrained checkpoint not found at {self.resume_path}")

#         self.model = GMFlowEstimator(
#             device=self.device,
#             resume=self.resume_path,
#             pred_bidir_flow=False,  # Test unidirectional flow
#             fwd_bwd_consistency_check=False,
#             padding_factor=16,
#             inference_size=None
#         )

#     def test_real_images_and_save_flow(self):
#         # Test the model on two real images and save the flow visualization
#         img1_path = './gmflow/demo/sintel_market_1/img1_batch.png'
#         img2_path = './gmflow/demo/sintel_market_1/img2_batch.png'
#         output_flow_path = './gmflow/flow_output.png'

#         # Check if image files exist
#         if not os.path.exists(img1_path) or not os.path.exists(img2_path):
#             self.skipTest(f"Image files not found: {img1_path}, {img2_path}")

#         # Load and preprocess the images
#         img1 = Image.open(img1_path).convert('RGB')
#         img2 = Image.open(img2_path).convert('RGB')
#         img1 = torch.tensor(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  # [1, 3, H, W]
#         img2 = torch.tensor(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  # [1, 3, H, W]
#         img1, img2 = img1.to(self.device), img2.to(self.device)

#         # Run the model
#         self.model.eval()  # Ensure evaluation mode
#         flow = self.model.forward(img1, img2)

#         # Check the output shape
#         self.assertEqual(flow.shape[0], 1, "Batch dimension should be 1")
#         self.assertEqual(flow.shape[1], 2, "Flow should have 2 channels (x, y)")
#         self.assertEqual(flow.shape[2], img1.shape[2], "Flow height should match input height")
#         self.assertEqual(flow.shape[3], img1.shape[3], "Flow width should match input width")

#         # Convert flow to [H, W, 2] format for visualization
#         flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 2]

#         # Save the flow visualization to a file
#         save_vis_flow_tofile(flow_np, output_flow_path)
#         self.assertTrue(os.path.exists(output_flow_path), f"Flow visualization file {output_flow_path} was not created")

#     def tearDown(self):
#         # Clean up VRAM
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# if __name__ == '__main__':
#     unittest.main()
# class TestGMFlowEstimator(unittest.TestCase):
#     def setUp(self):
#         # Set up a GMFlowEstimator instance with default parameters
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = GMFlowEstimator(device=self.device, resume='../pretrained/gmflow_things-e9887eda.pth')

#     def test_real_images_and_save_flow(self):
#         # Test the model on two real images and save the flow visualization
#         img1_path = '../demo/sintel_market_1/img1_batch.png'
#         img2_path = '../demo/sintel_market_1/img2_batch.png'
#         output_flow_path = './flow_output.png'  # Specify a file name for the flow visualization

#         # Load and preprocess the images
#         img1 = Image.open(img1_path).convert('RGB')
#         img2 = Image.open(img2_path).convert('RGB')
#         img1 = torch.tensor(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
#         img2 = torch.tensor(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
#         img1, img2 = img1.to(self.device), img2.to(self.device)

#         # Run the model
#         self.model.eval()  # Set to evaluation mode
#         self.model.pred_bidir_flow = False
#         flow = self.model.forward(img1, img2)
#         # Check the output shape
#         self.assertEqual(flow.shape[0], 1)  # Ensure batch dimension is present
#         self.assertEqual(flow.shape[1], 2)  # Ensure 2 channels for flow (x, y)

#         # Convert flow to [H, W, 2] format for visualization
#         flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 2]

#         # Save the flow visualization to a file
#         save_vis_flow_tofile(flow_np, output_flow_path)

# if __name__ == '__main__':
#     unittest.main()