'''
In this section we will look at how to persist model state with saving, loading and running model predictions.
'''

import torch
import torch.onnx as onnx
import torchvision.models as models

'''
PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method
'''

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

'''
To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.
'''

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

'''
When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass model (and not model.state_dict()) to the saving function:
'''
torch.save(model, 'model.pth')

'''
We can then load the model like this:
'''
model = torch.load('model.pth')

'''
Exporting Model to ONNX

PyTorch also has native ONNX export support. Given the dynamic nature of the PyTorch execution graph, however, the export process must traverse the execution graph to produce a persisted ONNX model. For this reason, a test variable of the appropriate size should be passed in to the export routine (in our case, we will create a dummy zero tensor of the correct size)
'''
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')

'''
There are a lot of things you can do with ONNX model, including running inference on different platforms and in different programming languages. For more details, we recommend visiting ONNX tutorial.

'''