import  {useContext} from 'react';
import { CssVarsProvider, extendTheme } from '@mui/joy/styles';
import GlobalStyles from '@mui/joy/GlobalStyles';
import CssBaseline from '@mui/joy/CssBaseline';
import Box from '@mui/joy/Box';
import IconButton from '@mui/joy/IconButton';
import Typography from '@mui/joy/Typography';
//import Stack from '@mui/joy/Stack';
import BadgeRoundedIcon from '@mui/icons-material/BadgeRounded';
//import { useLocation } from 'react-router-dom';
import icon from "../assets/4957136_4957136-Photoroom.png";
import icon2 from "../assets/20547283_6310507-Photoroom.png";
import { UserContext } from '../UserContext';
const customTheme = extendTheme({
  colorSchemes: {
    light: {},
    dark: {},
  },
});

export default function Dashboard() {
    const userContext = useContext(UserContext);

    if (!userContext) return null; // Avoid accessing undefined context
    const { user } = userContext;
  
  return (
    <CssVarsProvider theme={customTheme} defaultColorScheme="dark" disableTransitionOnChange>
      <CssBaseline />
      <GlobalStyles styles={{ ':root': { '--Form-maxWidth': '800px', '--Transition-duration': '0.4s' } }} />
      <Box sx={{ width: { xs: '100%', md: '50vw' }, position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'flex-end', backdropFilter: 'blur(12px)', backgroundColor: 'rgba(255 255 255 / 0.2)' }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100dvh', width: '100%', px: 2 }}>
          <Box component="header" sx={{ py: 3, display: 'flex', justifyContent: 'space-between' }}>
            <Box sx={{ gap: 2, display: 'flex', alignItems: 'center' }}>
              <IconButton variant="soft" color="primary" size="sm"><BadgeRoundedIcon /></IconButton>
              <Typography level="title-lg">Team VeinChain</Typography>
            </Box>
          </Box>
          <Box component="main" sx={{ my: 'auto', py: 2, pb: 5, display: 'flex', flexDirection: 'column', gap: 2, width: 400, maxWidth: '100%', mx: 'auto', borderRadius: 'sm', textAlign: 'center' }}>
            <Typography component="h1" level="h3">Welcome, {user || "guest"}!</Typography>
            <Typography level="body-md" sx={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'primary.main' }}>
              High Security Authentication Enabled
            </Typography>
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
        transitionDelay: 'calc(var(--Transition-duration) + 0.1s)',
        backdropFilter: 'blur(12px)',
        backgroundColor: 'rgba(255 255 255 / 0.2)',
        backgroundSize: 'cover', backgroundPosition: 'center', backgroundRepeat: 'no-repeat',
        backgroundImage: `url(${icon})`,
        [theme.getColorSchemeSelector('dark')]: {
          backgroundImage: `url(${icon2})`,
          backgroundColor: 'rgba(255 255 255 / 0.2)',
        }
      })} />
    </CssVarsProvider>
  );
}
