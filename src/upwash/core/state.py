class TrainingState:
    def __init__(self):
        self.global_step = 0

    def set_state(self, name: str, value):
        setattr(self, name, value)
        
    def get_state(self, name: str):
        return getattr(self, name)