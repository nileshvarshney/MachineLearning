import tensorflow as tf

######################################################
# Sample 1 creating graph
######################################################

g1 = tf.Graph()

with g1.as_default():
    with tf.Session() as sess:
        W = tf.constant([5,7],tf.int32, name= 'W')
        x = tf.placeholder(tf.int32, name = 'x')
        b = tf.constant([10,10], tf.int32, name = 'b')

        y = W * x + b

        print('Graph 1 result is :', sess.run(y , feed_dict={x:[2,3]}))
        assert y.graph is g1 # this will not resurn any error as g1 is default graph


######################################################
# Sample 2 creating graph and checking previous graph scopre
######################################################
g2 = tf.Graph()

with g2.as_default():
    with  tf.Session() as sess:
        W = tf.constant([5,7],tf.int32, name= 'W')
        x = tf.placeholder(tf.int32, name = 'x')
        y = W ** x
        print('Graph 2 result is :', sess.run(y , feed_dict={x:[2,3]}))
        # Below line will  return error as g1 is not default graph anymore
        # assert y.graph is g1
        assert y.graph is g2


######################################################
# Sample 3 by using default graph
######################################################
default_graph = tf.get_default_graph()
with  tf.Session() as sess:
    W = tf.constant([5,7],tf.int32, name= 'W')
    x = tf.placeholder(tf.int32, name = 'x')
    y = W + x
    print('Default Graph result is :', sess.run(y , feed_dict={x:[2,3]}))
    assert y.graph is default_graph


