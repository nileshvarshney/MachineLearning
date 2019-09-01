import tensorflow as tf

A = tf.constant([5], tf.int32, name = 'A')
B = tf.constant([5], tf.int32, name = 'B')
C = tf.constant([5], tf.int32, name = 'C')

x = tf.placeholder(tf.int32, name = 'A')

# Ax2 + Bx + C
with tf.name_scope("Equation_1"):
    AX1 = tf.multiply(A, tf.pow(x, 2), name = 'AX1')
    BX1 = tf.multiply(B, x, name = 'BX1')
    y1 = tf.add_n([AX1, BX1, C], name = 'y1')


# Ax2 + Bx + C
with tf.name_scope("Equation_2"):
    AX2 = tf.multiply(A, tf.pow(x, 2), name = 'AX2')
    BX2 = tf.multiply(B, tf.pow(x, 2) , name = 'BX2')
    y2 = tf.add(AX2, BX2, name = 'y1')

with tf.name_scope("Final_Calculation"):
    y =  y1 + y2

with tf.Session() as sess:
    print('Result  ==> ', sess.run(y, feed_dict={x:[13]}))

    # write to tensor board file
    writer = tf.summary.FileWriter('./namedScope',sess.graph)
    writer.close()