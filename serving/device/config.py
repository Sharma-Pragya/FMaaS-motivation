# DEVICE/config.py
import torch
import os

# Device selection
_cuda_device = os.environ.get("CUDA_DEVICE", None)
if _cuda_device:
    DEVICE = torch.device(_cuda_device)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default model directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Optional: limit GPU memory growth if needed
torch.backends.cudnn.benchmark = True

DECODERS={
    'mlp_momentsmall_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_momentbase_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_momentlarge_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':256,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronosmini_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':384,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronossmall_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronosbase_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronoslarge_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_papageis_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_papageip_ecgclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,       
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },

    'mlp_momentlarge_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_momentbase_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_momentsmall_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':256,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronosmini_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':384,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronossmall_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronosbase_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronoslarge_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_papageis_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_papageip_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,       
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_gestureclass':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':513*256,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronosmini_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':513*384,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronossmall_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':513*512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronosbase_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':513*768,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronoslarge_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':513*1024,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_momentlarge_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':64*1024,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_momentbase_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':64*768,'output_dim':192,'dropout':0.1}
        }       
    },
    
    'mlp_momentsmall_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':64*512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_papageis_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_papageip_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }
    },
    'mlp_papageissvri_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }
    },

    'mlp_momentlarge_illnessforecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':64*1024,'output_dim':36,'dropout':0.1}
        }       
    },

    'mlp_momentlarge_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_momentbase_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_momentsmall_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronostiny_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':256,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronosmini_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':384,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronossmall_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronosbase_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronoslarge_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_papageis_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_papageip_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,       
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },

    # Vision linear decoders (10-class EuroSAT)
    # DINOv2 variants
    'linear_dinosmall_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':384,'output_dim':10}}
    },
    'linear_dinobase_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':768,'output_dim':10}}
    },
    'linear_dinolarge_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1024,'output_dim':10}}
    },
    'linear_dinogiant_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1536,'output_dim':10}}
    },
    # Swin Transformer variants
    'linear_swintiny_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':768,'output_dim':10}}
    },
    'linear_swinsmall_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':768,'output_dim':10}}
    },
    'linear_swinbase_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1024,'output_dim':10}}
    },
    'linear_swinlarge_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1536,'output_dim':10}}
    },
    # MAE (ViT-MAE) variants
    'linear_maebase_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':768,'output_dim':10}}
    },
    'linear_maelarge_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1024,'output_dim':10}}
    },
    'linear_maehuge_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':1280,'output_dim':10}}
    },
    # VGG variants (4096-dim features after classifier head)
    'linear_vgg11_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':4096,'output_dim':10}}
    },
    'linear_vgg13_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':4096,'output_dim':10}}
    },
    'linear_vgg16_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':4096,'output_dim':10}}
    },
    'linear_vgg19_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':4096,'output_dim':10}}
    },
    # ResNet variants (pooler_output flattened: resnet18/34 = 512*7*7, resnet50/101 = 2048*7*7)
    'linear_resnet18_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':25088,'output_dim':10}}
    },
    'linear_resnet34_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':25088,'output_dim':10}}
    },
    'linear_resnet50_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':100352,'output_dim':10}}
    },
    'linear_resnet101_eurosatclass':{
        'decoder_type': 'linear',
        'decoder_config':{'DEVICE': DEVICE, 'cfg':{'input_dim':100352,'output_dim':10}}
    },
}

ADAPTERS = {
    'lora': {
        'adapter_type': 'lora',
        'adapter_config': {
            'r': 64,
            'lora_alpha': 32,
            'target_modules': ["q", "v"],
            'lora_dropout': 0.05,
        }
    },
    'lora_vlm': {
        'adapter_type': 'lora',
        'adapter_config': {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': ["q_proj", "v_proj"],
            'lora_dropout': 0.05,
        }
    },
}