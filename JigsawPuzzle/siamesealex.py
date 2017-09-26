import torch
import torch.nn as nn

class SiameseAlex(nn.Module):

    def __init__(self, num_classes=100):
        super(SiameseAlex, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(9 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), 2 * 2 * 256)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8, input9):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        output4 = self.forward_once(input4)
        output5 = self.forward_once(input5)
        output6 = self.forward_once(input6)
        output7 = self.forward_once(input7)
        output8 = self.forward_once(input8)
        output9 = self.forward_once(input9)
        concat_output = torch.cat((output1, output2, output3, output4, output5, output6, output7, output8, output9),1)
        output = self.classifier(concat_output)
        return output