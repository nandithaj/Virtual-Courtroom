module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",      // Ganache's default host
      port: 7545,             // Ganache's default port
      network_id: "5777",        // Match any network id
    //  from: "0x9f409a9B6497b34F2D1719198c127087d1e53424",  // Your Ganache account address
      gas: 6720,           // Gas limit (adjust if necessary)
     // gasPrice: 20000000000,  // Gas price (adjust if necessary)
    },
  },

  compilers: {
    solc: {
      version: "0.8.0",       // Use the version of Solidity you're using
    },
  },
};