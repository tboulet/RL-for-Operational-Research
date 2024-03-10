"""Memory module. This module implements memory objects for the agents to store their experiences and sample or use them for learning.
"""


class Memory:
    """Base class for memory objects. A memory object is used to store experiences and sample or use them for learning.
    """

    def __init__(self, config: Dict):
        """Initialize the memory object.

        Args:
            config (Dict): the configuration of the memory
        """
        self.config = config

    def store(self, experience: Tuple) -> None:
        """Store an experience in the memory.

        Args:
            experience (Tuple): the experience to store
        """
        raise NotImplementedError

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences from the memory.

        Args:
            batch_size (int): the size of the batch to sample

        Returns:
            List[Tuple]: the batch of experiences
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of experiences stored in the memory.

        Returns:
            int: the number of experiences stored in the memory
        """
        raise NotImplementedError