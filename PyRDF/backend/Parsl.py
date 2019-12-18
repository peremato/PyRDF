from PyRDF.backend.Dist import Dist
from PyRDF.backend.Utils import Utils
import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.threads import ThreadPoolExecutor
from parsl.app.app import python_app

@python_app
def app_map(mapper, x):
    return mapper(x)

@python_app
def app_reduce(reducer, l1, l2):
    return reducer(l1, l2)

class Parsl(Dist):
    """
    Backend that executes the computational graph using using `parsl` framework
    for distributed execution.
    """

    def __init__(self, config={}):
        """
        Creates an instance of the parsl backend class.

        Args:
            config (dict, optional): The config options for Parsl backend.
 
        Example::

            config = {
                'executor':'ThreadPoolExecutor'.
                'parameters':{'max_threads':16, 'label':'local_threads'}
                }
        """
        super(Parsl, self).__init__(config)

        _executor = globals()[config['executor']](**config['parameters'])
        _config = Config(executors=[_executor])

        parsl.clear()
        parsl.load(_config)
        self.parslConfig = _config

        self.npartitions = self._get_partitions()

    def _get_partitions(self):
        npart = (self.npartitions or
                 self.parslConfig.executors[0].max_threads or 4 )
        return int(npart)

    def ProcessAndMerge(self, mapper, reducer):
        """
        Performs map-reduce using parsl framework.

        Args:
            mapper (function): A function that runs the computational graph
                and returns a list of values.

            reducer (function): A function that merges two lists that were
                returned by the mapper.

        Returns:
            list: A list representing the values of action nodes returned
            after computation (Map-Reduce).
        """
        ranges = self.build_ranges()  # Get range pairs

        results = []
        for i in ranges:
            x = app_map(mapper, i)
            results.append(x)
        # reduce now the list of 'futures'
        while len(results) > 1:
            print('len(results)=',len(results))
            results.append(app_reduce(reducer,results.pop(0),results.pop(0)))
        return results.pop().result()

    def distribute_files(self, includes_list):
        """
        Spark supports sending files to the executors via the
        `SparkContext.addFile` method. This method receives in input the path
        to the file (relative to the path of the current python session). The
        file is initially added to the Spark driver and then sent to the
        workers when they are initialized.

        Args:
            includes_list (list): A list consisting of all necessary C++
                files as strings, created one of the `include` functions of
                the PyRDF API.
        """
        for filepath in includes_list:
            self.sparkContext.addFile(filepath)
