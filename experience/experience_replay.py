# what happens here is 
# the agent needs to know what it did in the similar 
# environment before so, the agent saves (state, action, new_state, terminated)
# into a memory similar to queue 

from collections import deque
import random

class ReplayMemory(): 
    def __init__(self, maxlen, seed = None): 
        self.memory = deque([], maxlen = maxlen)

        if seed is not None: 
            random.seed(seed)

    def append(self, transition): 
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)