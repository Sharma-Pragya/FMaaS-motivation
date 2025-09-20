components={
    'MOMENT-1-large':{'mem':4000},
    'MOMENT-1-base':{'mem':1000},
    'mlp_energyforecasting':{'mem':50},
    'mlp_weatherforecasting':{'mem':50},
    'energy_forecasting':{'mem':0}, #inference memory
    'weather_forecasting':{'mem':0}, #inference memory
}
pipelines={
    'p1':{'architecture':['MOMENT-1-large','mlp_energyforecasting','energy_forecasting'],'latency':0.001,'metric':0.9},
    # 'p3':{'architecture':['MOMENT-1-small','mlp_energyforecasting','energy_forecasting'],'latency':5,'metric':0.9},
    # 'p3':{'architecture':['MOMENT-1-small','mlp_weatherforecasting','weather_forecasting'],'latency':1,'metric':0.9},
    'p3':{'architecture':['MOMENT-1-large','mlp_weatherforecasting','weather_forecasting'],'latency':0.004,'metric':0.9},
}

tasks={'energy_forecasting':
       {
        'peak_workload':10,
        'latency':10,
        'metric':0.9
       },
       'weather_forecasting':{
        'peak_workload':10,
        'latency':10,
        'metric':0.9          
       }
       }