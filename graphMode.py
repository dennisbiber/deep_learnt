import tensorflow as tf

def eagerF(x):
	if x>0:
		x = x* x
	return x


@tf.function
def graphF(x):
	def if_true():
		return x * x
	def if_false():
		return x
	x = tf.cond(tf.greater(x, 0),
				if_true,
				if_false)
	return x


def eagerAdd(a, b):
	return a + b

@tf.function
def graphAdd(a, b):
	return a + b

v = tf.Variable(1.0)

with tf.GradientTape() as tape:
	result = graphAdd(v, 1.0)

def linear_layer(x):
	return 2*x + 1

@tf.function
def deep_net(x):
	return tf.nn.relu(linear_layer(x))

class CustomModel(tf.keras.models.Model):
	@tf.function
	def call(self, input_data):
		if tf.reduce_mean(input_data) > 0:
			return input_data
		else:
			return input_data // 2

@tf.function
def fizzbuzz(max_num):
	counter = 0
	for num in range(max_num):
		if num % 3 == 0 and num % 5 == 0:
			print("Fizzbuzz")
		elif num % 3 == 0:
			print("Fizz")
		elif num % 5 == 0:
			print("Buzz")
		else:
			print(num)
		counter += 1
	return counter

print(tf.autograph.to_code(fizzbuzz.python_function))

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
	a.assign(y *b)
	b.assign_add(x * a)
	return a + b

print(tf.autograph.to_code(f.python_function))

@tf.function
def blahF(x):
	while tf.reduce_sum(x) > 1:
		tf.print(x)
		x = tf.tanh(x)
	return x

print(tf.autograph.to_code(blahF.python_function))

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x,y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

print(f(1.0, 2.0))

print(tf.autograph.to_code(f.python_function))

@tf.function
def sign(x):
    if x > 0:
        return 'Positive'
    else:
        return 'Negative or zero'

print("Sign = {}".format(sign(tf.constant(2))))
print("Sign = {}".format(sign(tf.constant(-2))))

print(tf.autograph.to_code(sign.python_function))

@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(tf.autograph.to_code(sum_even.python_function))

def f(x):
    print("Traced with", x)

for i in range(5):
    f(2)
    
f(3)

@tf.function
def f(x):
    print("Traced with", x)

for i in range(5):
    f(2)
    
f(3)

@tf.function
def f(x):
    print("Traced with", x)
    # added tf.print
    tf.print("Executed with", x)

for i in range(5):
    f(2)
    
f(3)

def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

print(f(1))

@tf.function
def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

print(f(1))

# define the variables outside of the decorated function
v = tf.Variable(1.0)

@tf.function
def f(x):
    return v.assign_add(x)

print(f(5))