def success_response(
    data,
    message="Success"
):

    return {
        "success": True,
        "message": message,
        "data": data
    }



def error_response(
    error,
    message="Something went wrong"
):

    return {
        "success": False,
        "message": message,
        "error": error
    }