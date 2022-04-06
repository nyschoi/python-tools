class MyError(Exception):
    def __init__(self, msg="init_error_msg"):
        self.msg = msg

    def __str__(self):
        return self.msg


def input_cores():
    try:
        core_ = int(input("input your cores (max 10):"))
        if core_ > 10:
            raise MyError("Exceeded the limit!")
        print("Cores : ", core_)
    except MyError as e:
        print(e)
        input_cores()
    except ValueError as e:
        print(e)
        input_cores()


input_cores()
