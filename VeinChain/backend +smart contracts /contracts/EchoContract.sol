// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EchoContract {
    bytes public receivedData;

    function receiveData(bytes memory data) public {
        receivedData = data;
    }

    function getData() public view returns (bytes memory) {
        return receivedData;
    }
}
