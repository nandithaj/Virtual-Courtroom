import React, { useState } from 'react'
import axios from 'axios';


const SendData=()=>{
    const [name, setName]=useState('');
    const handleSend=async()=>{
       try {
           const response=await axios.post('http://localhost:8000/api/users', {name});
           console.log(response.data);
       } catch (error) {
           console.error('Error sending data:', error);
       }
    }
    return(
        <>
        <input type="text" placeholder="Enter your name" value={name} onChange={(e) => setName(e.target.value)} />
        <button onClick={handleSend}>Send</button>
        </>
    )
}

export default SendData;