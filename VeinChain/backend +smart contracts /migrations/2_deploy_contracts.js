const FingerVeinAuth = artifacts.require("FingerVeinAuth");

module.exports = function(deployer) {
  console.log("Starting deployment of FingerVeinAuth...");

  deployer.deploy(FingerVeinAuth,{ gas: 6721975 })
    .then(() => {
      console.log("FingerVeinAuth deployed successfully!");
      console.log("Contract Address:", FingerVeinAuth.address);
    })
    .catch((error) => {
      console.error("Deployment failed:", error);
    });
};
