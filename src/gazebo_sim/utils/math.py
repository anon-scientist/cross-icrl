class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
class Quaternion:
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

# Klasse zur Darstellung von Position und Rotation
class Transform:
    def __init__(self, position=None, rotation=None):
        self.position = position if position else Vector3()
        self.rotation = rotation if rotation else Quaternion()

    def __repr__(self):
        return f"Position: {self.position}, Rotation: {self.rotation}"