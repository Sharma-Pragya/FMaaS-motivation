# DEVICE/config.py
import torch
import os

# Device selection
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

    'mlp_momentlarge_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_momentbase_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_momentsmall_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':256,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_chronosmini_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':384,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_chronossmall_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_chronosbase_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':768,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_chronoslarge_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':1024,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_papageis_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':10,'hidden_dim':128}
        }
    },
    'mlp_papageip_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,       
            'cfg':{'input_dim':512,'output_dim':10,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'DEVICE': DEVICE,
            'cfg':{'input_dim':512,'output_dim':10,'hidden_dim':128}
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
    }
}