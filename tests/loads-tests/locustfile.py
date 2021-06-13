import time
from locust import HttpUser, task, between
from image_base64 import image_base64_str

class QuickstartUser(HttpUser):
    wait_time = between(2, 10)

    # @task
    # def hello_api(self):
    #     self.client.get("/")

    @task
    def face_swap(self):
        self.client.post("/api/v1/faceswap", data={"image_ref":image_base64_str, "template_name":"template3", "gender":"female"})
    
