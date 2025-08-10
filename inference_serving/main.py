from inference_serving.methods import proteus, our
import os
os.environ["GRB_LICENSE_FILE"] = "gurobi/gurobi.lic"
def main():
    # result=proteus.proteus()   
    # print("Proteus method executed successfully.")
    # print("Result:", result)

    result=our.our()   
    print("Our method executed successfully.")
    print("Result:", result)

if __name__ == "__main__":
    main()