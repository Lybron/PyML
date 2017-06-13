# Subclass of Node that will perform calculations and hold values

class Input(Node):
  def __init__(self):
    # An Input node has no inbound nodes
    # No need to pass anything to the Node instantiator
    Node.__init__(self)
  
  # Input node is the only node where the value may be passed as an argument to forward()
  # All other node implementations should get the value of previous node from self.inbound_nodes
  def forward(self, value=None):
    # Overwrite the value if one is passed in
    if value is not None:
      self.value = value