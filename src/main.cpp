#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "AgentWrapper.h"
#include <iostream>

const char* PyObjectToChar(PyObject* obj) {
    if (obj == nullptr) {
        return nullptr;
    }

    if (!PyUnicode_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Input object must be a Unicode string");
        return nullptr;
    }

    return PyUnicode_AsUTF8(obj);
}

PyObject* CharToPyObject(const char* value) {
    if (value == nullptr) {
        Py_RETURN_NONE;
    }

    return PyUnicode_FromString(value);
}


int main()
{

    AgentWrapper agent;

    // get the json from state_dicts
    auto [pNetJson, vNetJson] = agent.getStateDictsAsJson();

    // convert from PyObject* -> const char*
    const char *c_str = PyObjectToChar(pNetJson);
    if (c_str != NULL)
    {
        printf("Conversion successful!\n");
    }
    else
    {
        printf("Conversion failed or object is not a Unicode string.\n");
    }

    // std::cout << c_str << std::endl;

    // convert from const char* -> PyObject*
    PyObject *py_obj = CharToPyObject(c_str);
    if (py_obj != NULL)
    {
        // Do something with the Python object...

        Py_DECREF(py_obj); // Release the Python object
    }
    else
    {
        printf("Conversion from JSON string to Python object failed.\n");
    }

    std::cout << PyObjectToChar(py_obj) << std::endl;


    // get state_dicts from json
    auto [pStateDict, vStateDict] = agent.getStateDictsFromJson(pNetJson, vNetJson);
    // load state_dicts
    agent.loadStateDicts(pStateDict, vStateDict);

    // clean up
    // Py_DECREF(pStrObjFromJson);
    // Py_DECREF(pStrObj);

    return 0;
}
