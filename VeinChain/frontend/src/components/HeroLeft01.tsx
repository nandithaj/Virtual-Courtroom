/* eslint-disable jsx-a11y/anchor-is-valid */
import {useEffect,useRef} from 'react';
import Button from '@mui/joy/Button';
//import Link from '@mui/joy/Link';
import Typed from 'typed.js';
import Typography from '@mui/joy/Typography';
import ArrowForward from '@mui/icons-material/ArrowForward';
import TwoSidedLayout from './TwoSidedLayout';
import { useNavigate } from 'react-router-dom';

export default function HeroLeft01() {
  const navigate = useNavigate();
  const typedRef = useRef(null);

  useEffect(() => {
    const typed = new Typed(typedRef.current, {
      strings: ["VeinChain","Next-Gen Security"],
      typeSpeed: 50,
      backSpeed: 30,
      loop:true,
      showCursor: false,
    });

    return () => typed.destroy(); // Cleanup effect
  }, []);

  return (
    <TwoSidedLayout>
      <Typography color="primary" sx={{ fontSize: 'lg', fontWeight: 'lg' }}>
        Safe & Secure
      </Typography>
      <Typography
        level="h2"
        sx={{
          fontWeight: 'lg',
          fontSize: 'clamp(1.875rem, 1.3636rem + 2.1818vw, 3rem)',
        }}
      >
        
        <span ref={typedRef}></span>
        
      </Typography>
      <Typography
        textColor="text.secondary"
        sx={{ fontSize: 'lg', lineHeight: 'lg' }}
      >
        A Novel idea combining fingervein and BlockChain
      </Typography>
      <Button size="lg" onClick={()=>  navigate('/login')} endDecorator={<ArrowForward fontSize="large" />}>
        Get Started
      </Button>
      <Typography
        level="body-xs"
        sx={{
          position: 'absolute',
          top: '2rem',
          left: '50%',
          transform: 'translateX(-50%)',
        }}
      >
      
      </Typography>
    </TwoSidedLayout>
  );
}
