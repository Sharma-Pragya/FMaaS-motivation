class Request:
    def __init__(self, req_id, task, site_manager, device, backbone, req_time):
        self.req_id = req_id
        self.task = task
        self.site_manager = site_manager
        self.device = device
        self.backbone = backbone
        self.req_time = req_time

    def __repr__(self):
        return (f"req_id={self.req_id}, task={self.task}, "
                f"site_manager={self.site_manager}, device={self.device}, "
                f"backbone={self.backbone}, req_time={self.req_time}")

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "task": self.task,
            "site_manager": self.site_manager,
            "device": self.device,
            "backbone": self.backbone,
            "req_time": self.req_time,
        }
