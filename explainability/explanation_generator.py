from .gradcam import GradCAM

def generate_explanation(model, input_data):
    cam = GradCAM(model)
    return cam.generate(input_data)