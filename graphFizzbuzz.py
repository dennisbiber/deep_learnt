import tensorflow as tf

def tf__fizzbuzz(max_num):

    with ag__.FunctionScope('fizzbuzz', 
                            'fscope', 
                            ag__.ConversionOptions(recursive=True, 
                                                   user_requested=True, 
                                                   optional_features=(), 
                                                   internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()
        counter = 0

        def get_state_3():
            return (counter,)

        def set_state_3(vars_):
            (counter,) = vars_

        def loop_body(itr):
            num = itr

            def get_state_2():
                return ()

            def set_state_2(block_vars):
                pass

            def if_body_2():
                ag__.ld(print)('Fizzbuzz')

            def else_body_2():

                def get_state_1():
                    return ()

                def set_state_1(block_vars):
                    pass

                def if_body_1():
                    ag__.ld(print)('Fizz')

                def else_body_1():

                    def get_state():
                        return ()

                    def set_state(block_vars):
                        pass

                    def if_body():
                        ag__.ld(print)('Buzz')

                    def else_body():
                        ag__.ld(print)(ag__.ld(num))
                    ag__.if_stmt(((ag__.ld(num) % 5) == 0), if_body, else_body, get_state, set_state, (), 0)
                ag__.if_stmt(((ag__.ld(num) % 3) == 0), if_body_1, else_body_1, get_state_1, set_state_1, (), 0)
            ag__.if_stmt(ag__.and_((lambda : ((ag__.ld(num) % 3) == 0)), (lambda : ((ag__.ld(num) % 5) == 0))), if_body_2, else_body_2, get_state_2, set_state_2, (), 0)
            counter = ag__.ld(counter)
            counter += 1
            num = ag__.Undefined('num')
            ag__.for_stmt(ag__.converted_call(ag__.ld(range), (ag__.ld(max_num),), None, fscope), None, loop_body, get_state_3, set_state_3, ('counter',), {'iterate_names': 'num'})
        try:
            do_return = True
            retval_ = ag__.ld(counter)
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)


def tf__f(x, y):
    with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()
        ag__.converted_call(ag__.ld(a).assign, ((ag__.ld(y) * ag__.ld(b)),), None, fscope)
        ag__.converted_call(ag__.ld(b).assign_add, ((ag__.ld(x) * ag__.ld(a)),), None, fscope)
        try:
            do_return = True
            retval_ = (ag__.ld(a) + ag__.ld(b))
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)

a = tf.Variable(1.0)
b = tf.Variable(2.0)
print(tf__f(a, b))

def tf__blahF(x):
    with ag__.FunctionScope('blahF', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()

        def get_state():
            return (x,)

        def set_state(vars_):
            nonlocal x
            (x,) = vars_

        def loop_body():
            nonlocal x
            ag__.converted_call(ag__.ld(tf).print, (ag__.ld(x),), None, fscope)
            x = ag__.converted_call(ag__.ld(tf).tanh, (ag__.ld(x),), None, fscope)

        def loop_test():
            return (ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(x),), None, fscope) > 1)
        ag__.while_stmt(loop_test, loop_body, get_state, set_state, ('x',), {})
        try:
            do_return = True
            retval_ = ag__.ld(x)
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)