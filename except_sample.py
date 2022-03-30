class MyError(Exception):
    def __init__(self, msg="init_error_msg"):
        self.msg = msg

    def __str__(self):
        self.msg = '__str__ msg'
        return self.msg


def input_cores():
    try:
        core_ = input("input your cores (max 10):")
        if int(core_) > 10:
            # raise MyError("Exceeded the limit!")
            raise MyError
        print("Cores : ", core_)
    except MyError as e:
        print(e)
        input_cores()
    except ValueError as e:
        print(e)
        input_cores()


input_cores()