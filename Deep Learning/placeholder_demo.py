import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3],name ='y')

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name='prod_y')

final_div = tf.div(sum_x, prod_y, name='final_dev')
final_mean = tf.reduce_mean([sum_x, prod_y], name = 'final_mean')

sess = tf.Session()
print ("Sum :", sess.run(sum_x, feed_dict={x:[100, 200, 300]}))
print ("Product :", sess.run(prod_y, feed_dict={y:[1,2,3]}))

print ('Final Dev ', sess.run(final_div, feed_dict={x:[10, 20, 30], y: [1, 2, 3]}))
print ('Final Mean ', sess.run(final_mean, feed_dict={x:[1000, 2000, 3000], y: [10, 20, 30]}))

writer = tf.summary.FileWriter('./placeholder',sess.graph)
writer.close()
sess.close()

#####################################
# Example 2
#####################################

W = tf.constant([10,100], name='const_W')

x = tf.placeholder(tf.int32, name = 'x')
b = tf.placeholder(tf.int32, name = 'b')

Wx = tf.multiply(W, x, name = 'Wx')
y = tf.add(Wx, b, name = 'y') 
y_ = tf.subtract(x, b, name ="y_")

with tf.Session() as sess:
    print ('Intermediate Result Wx:', sess.run(Wx, feed_dict={x: [3, 15]}))
    print ('Final Result (Wx + b): ',sess.run(y, feed_dict={x:[5,100],b:[10,10]}))

    # calculate more than one value in a single go
    print ('Final multiple calcualtion (Wx + b) & (x -b)', sess.run(fetches = [y, y_], feed_dict={x:[5, 50], b :[7, 9]}))

#####################################
# Example 3, Variables
#####################################
W = tf.Variable([5.0, 10.0], tf.float32, name = 'W')
x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([12.0, 16.0], tf.float32, name = 'b')

y = W * x + b

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print ('Final Output  ( Wx + b)', sess.run(y, feed_dict={x :[10,20]}))

# it is not necessary to initialize all the valiables
y = W  * x
init = tf.variables_initializer([W])
with tf.Session() as sess:
    sess.run(init)
    print ('Intermediate Output  ( Wx )', sess.run(y, feed_dict={x :[10,20]}))


number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()
result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print('Result number * multiplier = ', sess.run(result))
        print('Increment the multiplier =', sess.run(multiplier.assign_add(1)))