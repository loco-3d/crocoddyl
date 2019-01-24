from crocoddyl import loadTalosArm, loadTalos, loadTalosLegs



if __name__ == "__main__":
    print("*** TALOS ARM ***")
    print(loadTalosArm().model)
    print("*** TALOS ARM floating ***")
    print(loadTalosArm(freeFloating=True).model)
    print("*** TALOS (floating) ***")
    print(loadTalos().model)
    print("*** TALOS LEGS (floating) ***")
    print(loadTalosLegs().model)