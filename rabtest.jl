using AMQPClient


port = AMQPClient.AMQP_DEFAULT_PORT

#login = AMQPClient.get_userid()  # default is usually "guest"
#password = get_password()  # default is usually "guest"
#auth_params = Dict{String,Any}("MECHANISM"=>"AMQPLAIN", "LOGIN"=>"guest", "PASSWORD"=>"guest")
amqps = amqps_configure()

conn = connection(; virtualhost="/", host="localhost", port=port, auth_params=AMQPClient.DEFAULT_AUTH_PARAMS)


# DO NOT USE THE AUTH PARAMS! 
                  #, amqps=amqps)

# create a message with 10 bytes of random value as data
msg =  Message(rand(UInt8, 10))
# create a persistent plain text message
data = convert(Vector{UInt8}, codeunits("hello world"))
msg = Message(data, content_type="text/plain", delivery_mode=PERSISTENT)

EXCG_DIRECT = "MyDirectExcg"
ROUTE1 = "routingkey1"
chan1 = channel(conn, AMQPClient.UNUSED_CHANNEL, true)
basic_publish(chan1, msg; exchange=EXCG_DIRECT, routing_key=ROUTE1)



