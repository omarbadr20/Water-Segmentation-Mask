import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """
    Two sequential layers of: [3x3 Conv] -> [Group Normalization] -> [ReLU].
    """
    def __init__(self, in_channels, out_channels, gn_groups=8):
        super(DoubleConv, self).__init__()
                
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

### ResNet50 Encoder + U-Net Decoder
class ResNet50UNet(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super(ResNet50UNet, self).__init__()
        
        # Load the ResNet-50 backbone

        self.encoder = models.resnet50(weights=None)
        
        # Declare first conv layer to accept 12 channels
        original_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
            
        # Extract features from the frozen ResNet backbone
        self.conv1_stem = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        
        # --- Decoder ---
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)  
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)   
        
        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(44, 32)    
        
        # Output Layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder 
        x_input = x                     
        
        x_conv1 = self.conv1_stem(x)    
        x_pool = self.maxpool(x_conv1)  
        
        x_layer1 = self.layer1(x_pool)  
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        
        # Decoder 
        u1 = self.up1(x_layer3)          
        u1 = torch.cat([u1, x_layer2], dim=1)
        u1 = self.conv_up1(u1)            
        
        u2 = self.up2(u1)                 
        u2 = torch.cat([u2, x_layer1], dim=1)
        u2 = self.conv_up2(u2)            
        
        u3 = self.up3(u2)                 
        u3 = torch.cat([u3, x_conv1], dim=1) 
        u3 = self.conv_up3(u3)            
        
        u4 = self.up4(u3)                 
        u4 = torch.cat([u4, x_input], dim=1)
        u4 = self.conv_up4(u4)            
        
        logits = self.outc(u4)            
        return logits