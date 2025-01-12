


class Endpoint:
    def __init__(self, name, addr):
        self.name = name
        self.addr = addr


    def get_name(self):
        return self.name
    
    
    def get_addr(self):
        return self.addr
    
    
    def get_scheme(self):   
        return self.addr.split("://")[0]
    
    
    def get_host(self):
        return self.addr.split("://")[1].split(":")[0]
    
    
    def get_port(self):
        return int(self.addr.split("://")[1].split(":")[1])
    

    def __str__(self):
        return f"{self.name}({self.addr})"

    def __repr__(self):
        return self.__str__()
    