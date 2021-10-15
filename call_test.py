from utils.data_process import multi_process

def f(s):
    return s

if __name__ == '__main__':
    print(multi_process(f, range(20), 5))