import os.path
from datetime import datetime
from enum import Enum
class LogLevel(Enum):
     INFO      = 1
     WARN      = 2
     ERROR     = 3
     HIGHLIGHT = 4
     HIDE      = 5

class Backlog():
    def __init__(self):
        self.files = {}
        self.folder = "results"
    def has_file(self, file:str) -> bool:
        return file in self.files.keys()
    def add_file(self, file:str) -> None:
        self.files[file] = []
        self.files[file].append(datetime.now().strftime("%d.%m.%Y, %H:%M:%S")+"\n")
    def write_to(self,file:str,msg:str) -> None:
        if not self.has_file(file):
            self.add_file(file)
        self.files[file].append(msg)
    def amount(self,file:str)-> int: 
        return len(self.files[file])
    def pop(self,file:str)-> None:
        path = os.path.join(self.folder, file)
        with open(path,"a") as f:
            for msg in self.files[file]:
                f.write(msg+"\n")
        self.files[file].clear()
    def pop_all(self) -> None:
        for file in self.files.keys():
            self.pop(file)

backlog = Backlog()

def log(*values:object, sep=' ', end='\n', file = None, level = LogLevel.INFO):
    msg = sep.join(str(v) for v in values)
    if level == LogLevel.INFO:
        print("\033[92mI:\033[00m",msg,sep='',end=end)
    elif level == LogLevel.WARN:
        print("\033[93mW:\033[00m",msg,sep='',end=end)
    elif level == LogLevel.ERROR:
        print("\033[91mE:\033[00m",msg,sep='',end=end)
    elif level == LogLevel.HIGHLIGHT:
        print("\033[96m",msg,"\033[00m",sep='',end=end)
    elif level == LogLevel.HIDE:
        pass
    if file is not None:
        backlog.write_to(file,"["+datetime.now().strftime("%H:%M:%S")+"] "+msg)
        if backlog.amount(file) > 20:
            backlog.pop(file)
    backlog.write_to("raw.txt",msg)
    if backlog.amount("raw.txt") > 1000:
        backlog.pop("raw.txt")