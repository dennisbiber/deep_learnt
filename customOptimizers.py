import tensorflow as tf

def fit_data(real_x, real_y):
	with tf.GradientTape(persistent=True) as tape:
		pred_y = w* real_x +b
		reg_loss = simple_loss(real_y, pred_y)

	w_gradient = tape.gradient(reg_loss, w)
	b_gradient = tape.gradient(reg_loss, b)

	w.assign_sub(w_gradient * LEARNING_RATE)
	b.assign_sub(b_gradient * LEARNING_RATE)


def train_step(images, labels):
	with tf.GradientTape() as tape:
		logits = model(images, training=True)
		loss_value = loss_object(labels, logits)

	loss_history.append(loss_value.numpy().mean())
	grads = tape.gradient(loss_value, model_trainable_variables)
	optimizer.apply.gradients(zip(grads, model_trainable_variables))

