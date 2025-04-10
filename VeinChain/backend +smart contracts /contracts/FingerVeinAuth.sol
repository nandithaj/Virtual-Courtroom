// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FingerVeinAuth {
    struct BiometricData {
        string encryptedFeatureVector;
        address owner;
    }

    mapping(string => BiometricData) public biometricTemplates;

    event UserEnrolled(string uid, address indexed owner);
    event UserAuthenticated(string uid, bool exists, bytes featureVector);

    function enrollUser(string memory uid, string memory encryptedFeatureVector) public {
        require(biometricTemplates[uid].owner == address(0), "User already enrolled");
        biometricTemplates[uid] = BiometricData(encryptedFeatureVector, msg.sender);
        emit UserEnrolled(uid, msg.sender);
    }

    function authenticateUser(string memory uid) public view returns (bool, string memory) {
        BiometricData memory storedData = biometricTemplates[uid];
        if (storedData.owner == address(0)) {
            return (false, "");
        }
        return (true, storedData.encryptedFeatureVector);
    }
}
