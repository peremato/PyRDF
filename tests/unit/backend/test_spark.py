import PyRDF, unittest, ROOT
from PyRDF.backend.Spark import Spark

class SelectionTest(unittest.TestCase):
    """
    A series of tests to check the
    accuracy of 'PyRDF.use' method.

    """

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up the `SparkContext` objects
        that were created during the tests in
        this class.

        """
        from pyspark import SparkContext
        context = SparkContext.getOrCreate()
        context.stop()

    def test_spark_select(self):
        """
        Test to check if 'spark'
        environment gets set correctly.

        """

        PyRDF.use("spark")
        self.assertIsInstance(PyRDF.current_backend, Spark)

class SparkBackendInitTest(unittest.TestCase):
    """
    Tests to ensure that the instance variables
    of `Spark` class are set according to the
    input `config` dict.

    """
    @classmethod
    def tearDown(cls):
        """
        Cleans up the `SparkContext` objects
        that were created.

        """
        from pyspark import SparkContext
        context = SparkContext.getOrCreate()
        context.stop()

    def test_set_spark_context_default(self):
        """
        Test case to check that if the config
        dictionary is empty, a `SparkContext`
        object is still created with default
        options for the current system.

        """
        from pyspark import SparkContext
        backend = Spark()

        self.assertDictEqual(backend.config, {})
        self.assertIsInstance(backend.sparkContext, SparkContext)

    def test_set_spark_context_with_conf(self):
        """
        Test case to check that a `SparkContext` object is
        correctly created for a given `SparkConf` object in
        the config dictionary.

        """
        from pyspark import SparkContext, SparkConf
        backend = Spark({'spark.app.name':'my-pyspark-app1'})

        self.assertIsInstance(backend.sparkContext, SparkContext)
        self.assertEqual(backend.sparkContext.getConf().get('spark.app.name'), 'my-pyspark-app1')

    def test_set_npartitions_explicit(self):
        """
        Test case to check that a `npartitions` is correctly
        set for a given input value in the config dictionary.

        """
        backend = Spark({"npartitions":5})

        self.assertEqual(backend.npartitions, 5)

    def test_npartitions_with_num_executors(self):
        """
        Test case to check that the value of `npartitions` is correctly set
        to number of executors when no input value is given in the config
        dictionary.

        """
        from pyspark import SparkContext, SparkConf
        backend = Spark({'spark.executor.instances':10})

        self.assertEqual(backend.npartitions, 10)

    def test_npartitions_with_already_existing_spark_context(self):
        """
        Test case to check that the value of `npartitions` is correctly set
        when a Spark Context already exists.

        """
        from pyspark import SparkContext, SparkConf
        sparkConf = SparkConf().set('spark.executor.instances', 15)
        sc = SparkContext(conf=sparkConf)
        backend = Spark()

        self.assertEqual(backend.npartitions, 15)

    def test_npartitions_default(self):
        """
        Test case to check that the default value of `npartitions` is
        correctly set when no input value is given in the config dictionary.

        """
        backend = Spark()

        self.assertEqual(backend.npartitions, 2)

class OperationSupportTest(unittest.TestCase):
    """
    Test cases to ensure that incoming operations are
    classified accurately in distributed environment.

    """
    @classmethod
    def tearDown(cls):
        """
        Cleans up the `SparkContext` objects
        that were created.

        """
        from pyspark import SparkContext
        context = SparkContext.getOrCreate()
        context.stop()

    def test_action(self):
        """
        Test case to check that action nodes
        are classified accurately.

        """
        # Check in Spark env
        backend = Spark()
        op = backend.check_supported("Histo1D")

    def test_transformation(self):
        """
        Test case to check that transformation
        nodes are classified accurately.

        """
        # Check in Spark env
        backend = Spark()
        op = backend.check_supported("Define")

    def test_unsupported_operations(self):
        """
        Test case to check that unsupported operations
        raise an Exception.

        """
        # Check in Spark env
        backend = Spark()
        with self.assertRaises(Exception):
            op = backend.check_supported("Take")

        with self.assertRaises(Exception):
            op = backend.check_supported("Snapshot")

        with self.assertRaises(Exception):
            op = backend.check_supported("Foreach")

        with self.assertRaises(Exception):
            op = backend.check_supported("Range")

    def test_none(self):
        """
        Test case to check that incorrect operations
        raise an Exception.

        """
        # Check in Spark env
        backend = Spark()
        with self.assertRaises(Exception):
            op = backend.check_supported("random")

    def test_range_operation_single_thread(self):
        """
        Test case to check that 'Range' operation
        works in single-threaded mode and raises an
        Exception in multi-threaded mode.

        """
       # Check in Spark env
        backend = Spark()
        with self.assertRaises(Exception):
            backend.check_supported("Range")

class InitializationTest(unittest.TestCase):
    """
    Check initialization method in the Spark backend

    """
    def test_initialization_method(self):
       # Define a method in the ROOT interpreter called getValue which
       # returns the value defined by the user on the python side
       def init(value):
           import ROOT
           cpp_code = '''int userValue = %s ;''' % value
           ROOT.gInterpreter.ProcessLine(cpp_code)

       PyRDF.initialize(init, 123)
       PyRDF.current_backend = Spark()
       # Spark backend has a limited list of supported methods, so we use Histo1D
       # which is a supported action.
       # The code below creates an RDataFrame instance with one single entry and
       # defines a column 'u' whose value is taken from the variable 'userValue'.
       # This variable is only declared inside the ROOT interpreter, however the
       # value of the variable is passed by the user from the python side.
       # If the init function defined by the user is properly propagated to the
       # Spark backend, each workers will run the init function as a first step
       # and hence the variable 'userValue' will be defined at runtime.
       # As a result the define operation should read the variable 'userValue'
       # and assign it to the entries of the column 'u' (only one entry).
       # Finally, Histo1D returns a histogram filled with one value. The mean
       # of this single value has to be the value itself, independently of
       # the number of spawned workers.
       df = PyRDF.RDataFrame(1).Define("u", "userValue").Histo1D("u")
       h = df.GetValue()
       self.assertEqual(h.GetMean(), 123)
