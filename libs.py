from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import torch