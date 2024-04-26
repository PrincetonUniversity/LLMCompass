class MemoryModule:
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity

memory_module_dict = {'A100_80GB': MemoryModule(80e9),'TPUv3': MemoryModule(float('inf')),'MI210': MemoryModule(64e9)}
