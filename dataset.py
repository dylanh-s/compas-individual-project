from abc import ABC, abstractmethod
import copy


class Dataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop('metadata', dict()) or dict()
        self.metadata.update({
            'transformer': '{}.__init__'.format(type(self).__name__),
            'params': kwargs,
            'previous': []
        })
        self.validate_dataset()

    def validate_dataset(self):
        """Error checking and type validation."""
        pass

    def copy(self, deepcopy=False):
        """
        Returns:
            Dataset: A new dataset with fields copied from this object and metadata.
        """
        cpy = copy.deepcopy(self) if deepcopy else copy.copy(self)
        # preserve any user-created fields
        cpy.metadata = cpy.metadata.copy()
        cpy.metadata.update({
            'transformer': '{}.copy'.format(type(self).__name__),
            'params': {'deepcopy': deepcopy},
            'previous': [self]
        })
        return cpy

    @abstractmethod
    def export_dataset(self):
        """Save this Dataset to disk."""
        raise NotImplementedError

    @abstractmethod
    def split(self, num_or_size_splits, shuffle=False):
        """Split this dataset into multiple partitions.

        Args:
            num_or_size_splits: floats < 1.0, take this proportions of the dataset.
            shuffle (bool, optional): Randomly shuffle the dataset before splitting.

        Returns:
            list(Dataset): Splits datasets depending on `num_or_size_splits`.
        """
        raise NotImplementedError
