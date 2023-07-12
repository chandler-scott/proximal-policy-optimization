#define PY_SSIZE_T_CLEAN
#include <Python.h>

int main()
{
    Py_Initialize();

    PyObject *pModule = PyImport_ImportModule("envs");
    if (pModule == NULL)
    {
        PyErr_Print();
        return 1;
    }

    PyObject *pTestWorld = PyObject_GetAttrString(pModule, "TestWorld");
    if (pTestWorld == NULL)
    {
        PyErr_Print();
        Py_DECREF(pModule);
        return 1;
    }

    Py_DECREF(pTestWorld);
    Py_DECREF(pModule);

    pModule = PyImport_ImportModule("gym");
    if (pModule == NULL)
    {
        PyErr_Print();
        return 1;
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "register");
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        if (PyErr_Occurred())
            PyErr_Print();
        return 1;
    }

    PyObject *pArgs = PyTuple_New(1);
    PyObject *pId = PyUnicode_FromString("TestWorld-v0");
    PyTuple_SetItem(pArgs, 0, pId);    

    PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
    if (pResult == NULL)
    {
        PyErr_Print();
        return 1;
    }

    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    // Rest of your code
    // ...

    Py_Finalize();
    return 0;
}
