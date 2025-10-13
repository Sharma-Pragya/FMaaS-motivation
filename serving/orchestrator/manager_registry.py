# orchestrator/manager_registry.py
import time

class SiteRegistry:
    def __init__(self):
        self.sites = {}  # site_name → {"url": str, "devices": list, "last_heartbeat": float}

    def register_site(self, site_name, url, devices=None):
        self.sites[site_name] = {
            "url": url,
            "devices": devices or [],
            "last_heartbeat": time.time(),
        }
        print(f"[Registry] Registered site: {site_name} ({url})")

    def update_heartbeat(self, site_name):
        if site_name in self.sites:
            self.sites[site_name]["last_heartbeat"] = time.time()

    def get_sites(self):
        return self.sites

site_registry = SiteRegistry()
