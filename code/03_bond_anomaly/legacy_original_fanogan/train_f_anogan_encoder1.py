import os
import sys
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset  # For custom dataset
import matplotlib.pyplot as plt
#import train_f_anogan1
#from fanogan.train_wgangp import train_wgangp
from fanogan.train_encoder_izif import train_encoder_izif

def main(opt):
    #if type(opt.seed) is int:
       # torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipeline = [transforms.Resize([opt.img_size]*2),
                transforms.RandomHorizontalFlip()]
    if opt.channels == 1:
        pipeline.append(transforms.Grayscale())
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

    transform = transforms.Compose(pipeline)
    # Define data transformations (adjust as needed)
    #pipeline = [transforms.Resize([opt.img_size]*2)]
    #if opt.channels == 1:
     #   pipeline.append(transforms.Grayscale())
    #pipeline.extend([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose(pipeline)

    # Custom dataset class for handling separate image and label files
    class AnomalyDataset(Dataset):
        def __init__(self, root_dir, transform):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".jpg")]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            # Assuming label files have same name (except extension)
            label_path = image_path.replace(".jpg", ".txt")
            image = Image.open(image_path).convert('RGB')  # Assuming RGB images
            image = self.transform(image)
            print(image)            
# Load label from text file
            label = torch.tensor(float(open(label_path).read()))
	    

            return image, label

    # Load your dataset
    dataset = AnomalyDataset(opt.train_root, transform)
    train_dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    print(train_dataloader)
    # ... rest of the code for model definition and training ...
    for images, labels in train_dataloader:
    # ... your training code here ...

    # Plot first few images (optional)
    	for i in range(min(len(images), 4)):  # Limit to 4 images
        	plt.imshow(images[i].permute(1, 2, 0).cpu())  # Permute and move to CPU
        	plt.title(f"Label: {labels[i]}")
        	plt.axis('off')
        	#plt.show()
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, device)
   # train_wgangp(opt, generator, discriminator, train_dataloader, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_root", type=str,
                        help="root name of your folder containing images and label files")
    # ... other arguments (unchanged) ...
#    opt = parser.parse_args()

####

#if __name__ == "__main__":
 #   import argparse
  #  parser = argparse.ArgumentParser()
   # parser.add_argument("train_root", type=str,
#                        help="root name of your dataset in train mode")
#    parser.add_argument("--force_download", "-f", action="store_true",
 #                       help="flag of force download")
    #parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    #parser.add_argument("--batch_size", type=int, default=28,
     #                   help="size of the batches")
   # parser.add_argument("--lr", type=float, default=0.0002,
#			help="root name of your dataset in train mode")
    parser.add_argument("--force_download", "-f", action="store_true", help="flag of force download")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=10,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    opt = parser.parse_args()
main(opt)


