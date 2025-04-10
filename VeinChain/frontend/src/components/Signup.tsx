import * as React from 'react';
import { CssVarsProvider, extendTheme, useColorScheme } from '@mui/joy/styles';
import GlobalStyles from '@mui/joy/GlobalStyles';
import CssBaseline from '@mui/joy/CssBaseline';
//import Box from '@mui/joy/Box';
import Button from '@mui/joy/Button';
import SvgIcon from '@mui/joy/SvgIcon';
import { styled } from '@mui/joy';
import Box from '@mui/joy/Box';
//import Typography from '@mui/joy/Typography';
import axios from 'axios';
//import Checkbox from '@mui/joy/Checkbox';
import Divider from '@mui/joy/Divider';
import FormControl from '@mui/joy/FormControl';
import FormLabel from '@mui/joy/FormLabel';
import IconButton, { IconButtonProps } from '@mui/joy/IconButton';
import Link from '@mui/joy/Link';
import Input from '@mui/joy/Input';
import Typography from '@mui/joy/Typography';
import Stack from '@mui/joy/Stack';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
//import BadgeRoundedIcon from '@mui/icons-material/BadgeRounded';
import Alert from '@mui/joy/Alert';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
//import GoogleIcon from '../GoogleIcon';
import { useRef, useState,useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import icon from "../assets/4957136_4957136-Photoroom.png";
import icon2 from "../assets/20547283_6310507-Photoroom.png";
import vein from "../assets/vein.png";
function ColorSchemeToggle(props: IconButtonProps) {
  const { onClick, ...other } = props;
  const { mode, setMode } = useColorScheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => setMounted(true), []);

  return (
    <IconButton
      aria-label="toggle light/dark mode"
      size="sm"
      variant="outlined"
      disabled={!mounted}
      onClick={(event) => {
        setMode(mode === 'light' ? 'dark' : 'light');
        onClick?.(event);
      }}
      {...other}
    >
      {mode === 'light' ? <DarkModeRoundedIcon /> : <LightModeRoundedIcon />}
    </IconButton>
  );
}
const VisuallyHiddenInput = styled('input')`
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  bottom: 0;
  left: 0;
  white-space: nowrap;
  width: 1px;
`;
const customTheme = extendTheme({
    colorSchemes: {
      light: {},
      dark: {},
    },
    //defaultColorScheme: { light: 'light', dark: 'dark' },
  });
  
export default function Login() {
  const [showCamera, setShowCamera] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const [username, setUsername] = useState("");
  //const [submittedData, setSubmittedData] = useState({});
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState("success");
  const navigate = useNavigate();
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);


  const handleOpenCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };
  const handleDeleteImage=() =>{
    setUploadedImage(null);
  };

  const handleCloseCamera = () => {
    setShowCamera(false);
    if (videoRef.current && videoRef.current.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
    }
  };
  const handleUploadImage = (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        setUploadedImage(file);  // Store the file directly
      }
    };
    
    const captureImage = () => {
      if (!videoRef.current) return;
    
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], "captured_image.png", { type: "image/png" });
          setCapturedImage(file);  // Store it directly as a File object
        }
      }, "image/png");
    };
    const handleSubmit = async () => {
      try {
        const formData = new FormData();
        formData.append("uid", username);
    
        if (capturedImage) {
          formData.append("image", capturedImage); // Directly send the File object
        } else if (uploadedImage) {
            formData.append("image",uploadedImage);
          
        }
    
        const response = await axios.post("https://01b0-2402-3a80-1cb9-e3d7-f8a3-19c-2c08-6083.ngrok-free.app/enroll", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        console.log(response.data);
        if (response.data.status==="User enrolled") {
          setMessage("User Enrollment Successful");
          setMessageType("success");
          //setUser(username);
          setTimeout(() => navigate("/login"), 2000);
        } else {
          setMessage("User Enrollment Failed");
          setMessageType("error");
        }
      } catch (error) {
        console.error("Upload Error:", error);
        setMessage("Error: Unable to enroll");
        setMessageType("error");
      }
      setTimeout(() => setMessage(""), 2000);
    };
    useEffect(() => {
       if (showCamera && videoRef.current && canvasRef.current) {
         const video = videoRef.current;
         const canvas = canvasRef.current;
         const ctx = canvas.getContext("2d");
     
         const drawGrid = () => {
           if (!ctx || !video.videoWidth || !video.videoHeight) return;
     
           // Set canvas size to match the video dimensions
           canvas.width = 640;
           canvas.height = 480;
     
           // Clear previous drawings
           ctx.clearRect(0, 0, canvas.width, canvas.height);
     
           // Draw the grid
           ctx.strokeStyle = "red";
           ctx.lineWidth = 1;
     
           // Number of rows and columns
           const cols = 10;
           const rows = 10;
           const cellWidth = canvas.width / cols;
           const cellHeight = canvas.height / rows;
     
           for (let i = 0; i <= cols; i++) {
             ctx.beginPath();
             ctx.moveTo(i * cellWidth, 0);
             ctx.lineTo(i * cellWidth, canvas.height);
             ctx.stroke();
           }
     
           for (let j = 0; j <= rows; j++) {
             ctx.beginPath();
             ctx.moveTo(0, j * cellHeight);
             ctx.lineTo(canvas.width, j * cellHeight);
             ctx.stroke();
           }
         };
     
         video.addEventListener("loadedmetadata", drawGrid);
         drawGrid();
     
         return () => {
           video.removeEventListener("loadedmetadata", drawGrid);
         };
       }
     }, [showCamera]);
    
  return (
    <CssVarsProvider theme={customTheme} defaultColorScheme="dark" disableTransitionOnChange>
      <CssBaseline />
      <GlobalStyles styles={{ ':root': { '--Form-maxWidth': '800px', '--Transition-duration': '0.5s' } }} />
      <Box sx={{ width: { xs: '100%', md: '50vw' }, position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'flex-end', backdropFilter: 'blur(12px)', backgroundColor: 'rgba(255 255 255 / 0.2)' }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100dvh', width: '100%', px: 2 }}>
          <Box component="header" sx={{ py: 3, display: 'flex', justifyContent: 'space-between' }}>
          <Box sx={{ gap: 2, display: 'flex', alignItems: 'center' }}>
          <img 
            src={vein}
            alt="Logo" 
            style={{ width: 32, height: 32 }} // Adjust size as needed
          />
          <Typography level="title-lg">Team VeinChain</Typography>
        </Box>
          <ColorSchemeToggle />
          </Box>
          <Box component="main" sx={{ my: 'auto', py: 2, pb: 5, display: 'flex', flexDirection: 'column', gap: 2, width: 400, maxWidth: '100%', mx: 'auto', borderRadius: 'sm' }}>
            <Stack sx={{ gap: 4, mb: 2 }}>
              <Stack sx={{ gap: 1 }}>
              {message && (
                <Alert variant='soft' color={messageType==="success"?"success":"danger"} startDecorator={messageType === "success" ? <CheckCircleOutlineIcon /> : <ErrorOutlineIcon />}  sx={{ fontWeight: 'md', textAlign: 'center' }}>
                  {message}
                </Alert>
              )}
                <Typography component="h1" level="h3">Sign Up</Typography>
                <Typography level="body-sm">
                  Already a user?{' '}
                  <Link href="" level="title-sm" onClick={() => navigate('/login')}>Log in!</Link>
                </Typography>
              </Stack>
            </Stack>
            <FormControl required>
              <FormLabel>Username</FormLabel>
              <Input type="text" name="email" value={username} onChange={(e) => setUsername(e.target.value)}/>
            </FormControl>
            <Button variant="outlined" onClick={handleOpenCamera}>Camera For Vein Pattern</Button>
            {showCamera ? (
              <Box sx={{ textAlign: 'center' }}>
                <Typography level="body-md">Vein Pattern Authentication</Typography>
                <div style={{ position: "relative", display: "inline-block" }}>
    {/* Video feed */}
    <video ref={videoRef} autoPlay className="rounded mb-4" style={{ width: "100%" }} />

    {/* Canvas overlay */}
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none", // Prevent interaction
      }}
    />
  </div>

                <Stack direction="row" spacing={2} justifyContent="center">
                  <Button onClick={captureImage}>Capture</Button>
                  <Button onClick={handleCloseCamera}>Close</Button>
                </Stack>
                {capturedImage && <img src={URL.createObjectURL(capturedImage)} alt="Captured" style={{ marginTop: '10px', maxWidth: '100%' }} />}
              </Box>
            ):
            (
              <Box sx={{ textAlign: 'center' }}>
              <Stack spacing={2}>
              <Divider>Or</Divider>
              
              <Button
                component="label"
                role={undefined}
                tabIndex={-1}
                variant="outlined"
                color="neutral"
                startDecorator={
                  <SvgIcon>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"
                      />
                    </svg>
                  </SvgIcon>
                }
              >
                Upload a file
                <VisuallyHiddenInput type="file" accept="image/*" onChange={handleUploadImage} />
              </Button>
               
              {uploadedImage && <img src={URL.createObjectURL(uploadedImage)} alt="Captured" style={{ marginTop: '10px', maxWidth: '100%' }} />}
             {uploadedImage && <Button onClick={handleDeleteImage}>Close</Button>}
             </Stack>
            </Box>
            
                      )}
              
            <Stack spacing={2}>
            <Button variant="solid" onClick={handleSubmit} disabled={!username || !(capturedImage||uploadedImage)}>Submit</Button>
            
            </Stack>
          </Box>
          <Box component="footer" sx={{ py: 3 }}>
            <Typography level="body-xs" sx={{ textAlign: 'center' }}>
              © VeinChain {new Date().getFullYear()}
            </Typography>
          </Box>
        </Box>
      </Box>
      <Box sx={(theme) => ({
        height: '100%', position: 'fixed', right: 0, top: 0, bottom: 0, left: { xs: 0, md: '50vw' },
        transition: 'background-image var(--Transition-duration), left var(--Transition-duration) !important',
        transitionDelay: 'calc(var(--Transition-duration) + 0.5s)',
        backdropFilter: 'blur(12px)',
        backgroundColor: 'rgba(255 255 255 / 0.5)',
       backgroundSize: 'cover', backgroundPosition: 'center', backgroundRepeat: 'no-repeat',
        backgroundImage: `url(${icon})`,
        [theme.getColorSchemeSelector('dark')]: {
          backgroundImage: `url(${icon2})`,
          backgroundColor:'rgba(255 255 255 / 0.2)',
        }
      })} />
    </CssVarsProvider>
  );
}
